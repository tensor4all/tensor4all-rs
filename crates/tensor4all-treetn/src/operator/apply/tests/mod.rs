mod identity_metadata;

use super::*;
use crate::random::{random_treetn, LinkSpace};
use crate::treetn::contraction::SrcOptions;
use crate::{
    LinearOperatorIndexBindingError, LinearOperatorTaggedApplyError, NumberedTagSelectionError,
    SiteIndexNetwork,
};
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::{ColMajorArrayRef, IdxTensor, SvdTruncationPolicy, TensorContractionLike};
use tensor4all_tensorbackend::StorageKind;

type DynIndex = Index<DynId, TagSet>;
type ChainStateAndIdentity = (
    TreeTN<IdxTensor, String>,
    LinearOperator<IdxTensor, String>,
    Vec<(String, DynIndex)>,
);

fn make_index(dim: usize) -> DynIndex {
    Index::new_dyn(dim)
}

fn make_mapping() -> IndexMapping<DynIndex> {
    IndexMapping {
        true_index: make_index(2),
        internal_index: make_index(2),
    }
}

fn build_identity_operator(sites: &[(String, DynIndex)]) -> LinearOperator<IdxTensor, String> {
    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for (name, true_index) in sites {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let tensor = IdxTensor::from_dense(
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

fn build_bonded_identity_operator(
    sites: &[(String, DynIndex)],
) -> LinearOperator<IdxTensor, String> {
    assert_eq!(sites.len(), 2);

    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let bond = make_index(1);

    for (name, true_index) in sites {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let tensor = if name == &sites[0].0 {
            IdxTensor::from_dense(
                vec![
                    internal_output.clone(),
                    internal_input.clone(),
                    bond.clone(),
                ],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap()
        } else {
            IdxTensor::from_dense(
                vec![
                    bond.clone(),
                    internal_output.clone(),
                    internal_input.clone(),
                ],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap()
        };

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

    let n0 = mpo.node_index(&sites[0].0).unwrap();
    let n1 = mpo.node_index(&sites[1].0).unwrap();
    mpo.connect(n0, &bond, n1, &bond).unwrap();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn build_redundant_bonded_identity_operator(
    sites: &[(String, DynIndex)],
    mpo_bond_dim: usize,
) -> LinearOperator<IdxTensor, String> {
    assert_eq!(sites.len(), 2);

    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let bond = make_index(mpo_bond_dim);

    for (site_pos, (name, true_index)) in sites.iter().enumerate() {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let dim = true_index.dim();
        let tensor = if site_pos == 0 {
            let mut data = vec![0.0; dim * dim * mpo_bond_dim];
            for bond_value in 0..mpo_bond_dim {
                for site_value in 0..dim {
                    data[site_value + dim * (site_value + dim * bond_value)] = 1.0;
                }
            }
            IdxTensor::from_dense(
                vec![
                    internal_output.clone(),
                    internal_input.clone(),
                    bond.clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            let mut data = vec![0.0; mpo_bond_dim * dim * dim];
            for bond_value in 0..mpo_bond_dim {
                for site_value in 0..dim {
                    data[bond_value + mpo_bond_dim * (site_value + dim * site_value)] =
                        1.0 / mpo_bond_dim as f64;
                }
            }
            IdxTensor::from_dense(
                vec![
                    bond.clone(),
                    internal_output.clone(),
                    internal_input.clone(),
                ],
                data,
            )
            .unwrap()
        };

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

    let n0 = mpo.node_index(&sites[0].0).unwrap();
    let n1 = mpo.node_index(&sites[1].0).unwrap();
    mpo.connect(n0, &bond, n1, &bond).unwrap();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn build_chain_state() -> (TreeTN<IdxTensor, String>, Vec<(String, DynIndex)>) {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let s3 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);
    let b23 = make_index(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        IdxTensor::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 =
        IdxTensor::from_dense(vec![b12.clone(), s2.clone(), b23.clone()], vec![1.0; 8]).unwrap();
    let t3 = IdxTensor::from_dense(vec![b23.clone(), s3.clone()], vec![1.0; 4]).unwrap();

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

#[test]
fn apply_linear_operator_supports_multiple_index_mappings_per_node() {
    let x = make_index(2);
    let y = make_index(2);
    let state_tensor =
        IdxTensor::from_dense(vec![x.clone(), y.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["site".to_string()])
            .unwrap();

    let x_in = make_index(2);
    let y_in = make_index(2);
    let x_out = make_index(2);
    let y_out = make_index(2);
    let mut data = vec![0.0; 16];
    for xi in 0..2 {
        for yi in 0..2 {
            let xo = xi;
            let yo = yi;
            data[xo + 2 * (yo + 2 * (xi + 2 * yi))] = 1.0;
        }
    }
    let mpo_tensor = IdxTensor::from_dense(
        vec![x_out.clone(), y_out.clone(), x_in.clone(), y_in.clone()],
        data,
    )
    .unwrap();
    let mpo = TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["site".to_string()])
        .unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "site".to_string(),
        vec![
            IndexMapping {
                true_index: x.clone(),
                internal_index: x_in,
            },
            IndexMapping {
                true_index: y.clone(),
                internal_index: y_in,
            },
        ],
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "site".to_string(),
        vec![
            IndexMapping {
                true_index: x.clone(),
                internal_index: x_out,
            },
            IndexMapping {
                true_index: y.clone(),
                internal_index: y_out,
            },
        ],
    );

    let operator = LinearOperator::new_multi(mpo, input_mapping, output_mapping);
    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1e-12
    );
}

#[test]
fn apply_linear_operator_to_indices_binds_explicit_external_indices() {
    let state_index = make_index(2);
    let state_tensor = IdxTensor::from_dense(vec![state_index.clone()], vec![2.0, 5.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["site".to_string()])
            .unwrap();

    let op_true_input = make_index(2);
    let op_true_output = make_index(2);
    let internal_input = make_index(2);
    let internal_output = make_index(2);
    let mpo_tensor = IdxTensor::from_dense(
        vec![internal_output.clone(), internal_input.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let mpo = TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["site".to_string()])
        .unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "site".to_string(),
        IndexMapping {
            true_index: op_true_input.clone(),
            internal_index: internal_input,
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "site".to_string(),
        IndexMapping {
            true_index: op_true_output.clone(),
            internal_index: internal_output,
        },
    );
    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    let result = apply_linear_operator_to_indices(
        &operator,
        &state,
        &[(op_true_input, state_index.clone())],
        &[(op_true_output, state_index.clone())],
        ApplyOptions::naive(),
    )
    .unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1e-12
    );
}

#[test]
fn bind_linear_operator_indices_reports_typed_error_paths() {
    let op_true_input = make_index(2);
    let operator = build_identity_operator(&[("site".to_string(), op_true_input.clone())]);
    let same_id_prime = op_true_input.prime();

    let missing = bind_linear_operator_indices(&operator, &[(same_id_prime, make_index(2))], &[])
        .unwrap_err();
    assert!(matches!(
        missing,
        LinearOperatorIndexBindingError::MissingSourceIndex { role: "input", .. }
    ));

    let mismatch =
        bind_linear_operator_indices(&operator, &[(op_true_input.clone(), make_index(3))], &[])
            .unwrap_err();
    assert!(matches!(
        mismatch,
        LinearOperatorIndexBindingError::DimensionMismatch {
            role: "input",
            old_dim: 2,
            new_dim: 3,
        }
    ));

    let duplicate = bind_linear_operator_indices(
        &operator,
        &[
            (op_true_input.clone(), make_index(2)),
            (op_true_input, make_index(2)),
        ],
        &[],
    )
    .unwrap_err();
    assert!(matches!(
        duplicate,
        LinearOperatorIndexBindingError::DuplicateSourceIndex { role: "input", .. }
    ));
}

#[test]
fn apply_linear_operator_to_numbered_tags_binds_state_indices_in_tag_order() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let k2 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    let bond = make_index(1);
    let state = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![k1.clone(), bond.clone()], vec![1.0, 2.0]).unwrap(),
            IdxTensor::from_dense(vec![bond, k2.clone()], vec![3.0, 5.0]).unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();
    let operator = build_bonded_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ]);

    let result =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1.0e-12
    );
}

#[test]
fn apply_linear_operator_to_numbered_tags_reports_missing_tag() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let state = TreeTN::<IdxTensor, String>::from_tensors(
        vec![IdxTensor::from_dense(vec![k1], vec![1.0, 2.0]).unwrap()],
        vec!["site0".to_string()],
    )
    .unwrap();
    let operator = build_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ]);

    let err =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap_err();

    assert!(matches!(
        err,
        LinearOperatorTaggedApplyError::SelectTaggedIndices(
            NumberedTagSelectionError::MissingTag { ref tag }
        ) if tag == "k=2"
    ));
}

/// Build a product-state chain `site0..site{length-1}` whose site `i` carries
/// the numbered tag `k=(i+1)`.
fn build_numbered_tag_product_chain(
    length: usize,
) -> (TreeTN<IdxTensor, String>, Vec<(String, DynIndex)>) {
    let site_indices: Vec<DynIndex> = (0..length)
        .map(|i| {
            Index::new_dyn_with_tags(2, TagSet::from_str(&format!("Qubit,k={}", i + 1)).unwrap())
        })
        .collect();
    let bonds: Vec<DynIndex> = (0..length.saturating_sub(1))
        .map(|_| make_index(1))
        .collect();

    let mut tensors = Vec::with_capacity(length);
    let mut names = Vec::with_capacity(length);
    for i in 0..length {
        let mut legs = Vec::new();
        if i > 0 {
            legs.push(bonds[i - 1].clone());
        }
        legs.push(site_indices[i].clone());
        if i + 1 < length {
            legs.push(bonds[i].clone());
        }
        let len: usize = legs.iter().map(|index| index.dim()).product();
        let mut data = vec![0.0; len];
        // All bond dimensions are 1, so the site leg has stride 1.
        data[0] = 1.0;
        data[1] = (i as f64) + 2.0;
        tensors.push(IdxTensor::from_dense(legs, data).unwrap());
        names.push(format!("site{i}"));
    }

    let state = TreeTN::from_tensors(tensors, names.clone()).unwrap();
    let sites = names.into_iter().zip(site_indices).collect();
    (state, sites)
}

/// Build a chain MPO with the same node names as `sites`, identity everywhere
/// except `factor_node`, where `factor` (output rows, input columns) is used.
/// Returns the operator plus each node's distinct true input/output indices.
#[allow(clippy::type_complexity)]
fn build_numbered_tag_chain_operator(
    sites: &[(String, DynIndex)],
    factor_node: &str,
    factor: [[f64; 2]; 2],
) -> (
    LinearOperator<IdxTensor, String>,
    Vec<(String, DynIndex, DynIndex)>,
) {
    let length = sites.len();
    let mpo_bonds: Vec<DynIndex> = (0..length.saturating_sub(1))
        .map(|_| make_index(1))
        .collect();

    let mut tensors = Vec::with_capacity(length);
    let mut names = Vec::with_capacity(length);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let mut true_indices = Vec::with_capacity(length);

    for (pos, (name, _state_site)) in sites.iter().enumerate() {
        let true_input = make_index(2);
        let true_output = make_index(2);
        let internal_input = make_index(2);
        let internal_output = make_index(2);

        let mut legs = Vec::new();
        if pos > 0 {
            legs.push(mpo_bonds[pos - 1].clone());
        }
        legs.push(internal_output.clone());
        legs.push(internal_input.clone());
        if pos + 1 < length {
            legs.push(mpo_bonds[pos].clone());
        }
        let len: usize = legs.iter().map(|index| index.dim()).product();
        let mut data = vec![0.0; len];
        let matrix = if name == factor_node {
            factor
        } else {
            [[1.0, 0.0], [0.0, 1.0]]
        };
        // Column-major [bond_prev?, out, in, bond_next?]; both bond dims are 1,
        // so the out leg has stride 1 and the in leg stride 2.
        for out in 0..2 {
            for input in 0..2 {
                data[out + 2 * input] = matrix[out][input];
            }
        }
        tensors.push(IdxTensor::from_dense(legs, data).unwrap());
        names.push(name.clone());
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_input.clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_output.clone(),
                internal_index: internal_output,
            },
        );
        true_indices.push((name.clone(), true_input, true_output));
    }

    let mpo = TreeTN::from_tensors(tensors, names).unwrap();
    (
        LinearOperator::new(mpo, input_mapping, output_mapping),
        true_indices,
    )
}

#[test]
fn apply_linear_operator_to_numbered_tags_binds_by_node_identity_for_long_chains() {
    // With 11 nodes, lexicographic node order (`site0, site1, site10, site2, ...`)
    // differs from tag order. A local X on `site10` must act on `site10`.
    let (state, sites) = build_numbered_tag_product_chain(11);
    let factor = [[0.0, 1.0], [1.0, 0.0]];
    let (operator, true_indices) = build_numbered_tag_chain_operator(&sites, "site10", factor);

    for options in [ApplyOptions::naive(), ApplyOptions::default()] {
        let result =
            apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, options.clone())
                .unwrap();

        let mut input_pairs = Vec::new();
        let mut output_pairs = Vec::new();
        for ((_, state_index), (_, true_input, true_output)) in
            sites.iter().zip(true_indices.iter())
        {
            input_pairs.push((true_input.clone(), state_index.clone()));
            output_pairs.push((true_output.clone(), state_index.clone()));
        }
        let expected = apply_linear_operator_to_indices(
            &operator,
            &state,
            &input_pairs,
            &output_pairs,
            options.clone(),
        )
        .unwrap();

        let error = result
            .to_dense()
            .unwrap()
            .sub(&expected.to_dense().unwrap())
            .unwrap()
            .maxabs()
            .unwrap();
        assert!(
            error < 1.0e-10,
            "node-identity binding failed for {:?}: maxabs error {error}",
            options.method
        );
    }
}

#[test]
fn apply_linear_operator_to_numbered_tags_respects_within_node_tag_order() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let k2 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    // The node tensor stores its site legs as [k2, k1], the reverse of tag order.
    let state_tensor =
        IdxTensor::from_dense(vec![k2.clone(), k1.clone()], vec![3.0, 5.0, 6.0, 10.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["site".to_string()])
            .unwrap();
    assert_eq!(
        state.node_site_indices(&"site".to_string()),
        Some(vec![k2.clone(), k1.clone()])
    );

    let true_in0 = make_index(2);
    let true_in1 = make_index(2);
    let true_out0 = make_index(2);
    let true_out1 = make_index(2);
    let in0 = make_index(2);
    let in1 = make_index(2);
    let out0 = make_index(2);
    let out1 = make_index(2);
    // Identity on mapping 0 and X on mapping 1, in column-major
    // [out0, out1, in0, in1] leg order.
    let mut data = vec![0.0; 16];
    for o0 in 0..2 {
        for o1 in 0..2 {
            for i0 in 0..2 {
                for i1 in 0..2 {
                    let identity = if o0 == i0 { 1.0 } else { 0.0 };
                    let x = if o1 + i1 == 1 { 1.0 } else { 0.0 };
                    data[o0 + 2 * o1 + 4 * i0 + 8 * i1] = identity * x;
                }
            }
        }
    }
    let mpo_tensor = IdxTensor::from_dense(
        vec![out0.clone(), out1.clone(), in0.clone(), in1.clone()],
        data,
    )
    .unwrap();
    let mpo = TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["site".to_string()])
        .unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "site".to_string(),
        vec![
            IndexMapping {
                true_index: true_in0.clone(),
                internal_index: in0,
            },
            IndexMapping {
                true_index: true_in1.clone(),
                internal_index: in1,
            },
        ],
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "site".to_string(),
        vec![
            IndexMapping {
                true_index: true_out0.clone(),
                internal_index: out0,
            },
            IndexMapping {
                true_index: true_out1.clone(),
                internal_index: out1,
            },
        ],
    );
    let operator = LinearOperator::new_multi(mpo, input_mapping, output_mapping);

    let result =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap();
    // Tag order binds k=1 to mapping 0 and k=2 to mapping 1, independent of the
    // node tensor's [k2, k1] leg order.
    let expected = apply_linear_operator_to_indices(
        &operator,
        &state,
        &[(true_in0, k1.clone()), (true_in1, k2.clone())],
        &[(true_out0, k1), (true_out1, k2)],
        ApplyOptions::naive(),
    )
    .unwrap();

    let error = result
        .to_dense()
        .unwrap()
        .sub(&expected.to_dense().unwrap())
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(error < 1.0e-12, "within-node tag order mismatch: {error}");
}

#[test]
fn apply_linear_operator_to_numbered_tags_reports_missing_node_mapping() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let k2 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    let bond = make_index(1);
    let state = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![k1, bond.clone()], vec![1.0, 2.0]).unwrap(),
            IdxTensor::from_dense(vec![bond, k2], vec![3.0, 5.0]).unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();

    // Operator mappings live on different node names, so no selected index finds
    // a mapping on its own node.
    let operator = build_identity_operator(&[
        ("other0".to_string(), make_index(2)),
        ("other1".to_string(), make_index(2)),
    ]);
    let err =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap_err();
    assert!(matches!(
        err,
        LinearOperatorTaggedApplyError::MissingNodeMapping {
            role: "input",
            ref tag,
            ..
        } if tag == "k=1"
    ));

    // Input mappings on the state nodes but an output mapping keyed elsewhere
    // exercises the output-side missing-node path.
    let mut input_mapping = HashMap::new();
    input_mapping.insert("site0".to_string(), vec![make_mapping()]);
    input_mapping.insert("site1".to_string(), vec![make_mapping()]);
    let mut output_mapping = HashMap::new();
    output_mapping.insert("site0".to_string(), vec![make_mapping()]);
    output_mapping.insert("other".to_string(), vec![make_mapping()]);
    let mpo = build_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ])
    .mpo()
    .clone();
    let operator = LinearOperator::new_multi(mpo, input_mapping, output_mapping);
    let err =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap_err();
    assert!(matches!(
        err,
        LinearOperatorTaggedApplyError::MissingNodeMapping {
            role: "output",
            ref tag,
            ..
        } if tag == "k=2"
    ));
}

#[test]
fn apply_linear_operator_to_numbered_tags_reports_per_node_count_mismatch() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let k2 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    // Both numbered tags live on `site0`; `site1` carries no numbered tag.
    let bond = make_index(1);
    let state = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![k2, k1, bond.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            IdxTensor::from_dense(vec![bond, make_index(2)], vec![5.0, 6.0]).unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();

    // One mapping per node: `site0` declares one mapping but receives two tags.
    let operator = build_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ]);
    let err =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap_err();
    assert!(
        matches!(
            err,
            LinearOperatorTaggedApplyError::SelectedTagCountMismatch { role: "input", .. }
        ),
        "{err:?}"
    );

    // Input counts match the selected tags while output counts do not, which
    // exercises the output-side per-node count check.
    let k3 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=3").unwrap());
    let bond = make_index(1);
    let state = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![
                    Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap()),
                    Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap()),
                    bond.clone(),
                ],
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, k3], vec![5.0, 6.0]).unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert("site0".to_string(), vec![make_mapping(), make_mapping()]);
    input_mapping.insert("site1".to_string(), vec![make_mapping()]);
    let mut output_mapping = HashMap::new();
    output_mapping.insert("site0".to_string(), vec![make_mapping()]);
    output_mapping.insert("site1".to_string(), vec![make_mapping(), make_mapping()]);
    let mpo = build_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ])
    .mpo()
    .clone();
    let operator = LinearOperator::new_multi(mpo, input_mapping, output_mapping);
    let err =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap_err();
    assert!(
        matches!(
            err,
            LinearOperatorTaggedApplyError::SelectedTagCountMismatch { role: "output", .. }
        ),
        "{err:?}"
    );
}

fn build_tree_state() -> (TreeTN<IdxTensor, String>, Vec<(String, DynIndex)>) {
    let s_a = make_index(2);
    let s_b = make_index(2);
    let s_c = make_index(2);
    let s_d = make_index(2);
    let s_e = make_index(2);
    let b_ab = make_index(2);
    let b_bc = make_index(2);
    let b_bd = make_index(2);
    let b_be = make_index(2);

    let t_a = IdxTensor::from_dense(vec![s_a.clone(), b_ab.clone()], vec![1.0; 4]).unwrap();
    let t_b = IdxTensor::from_dense(
        vec![b_ab.clone(), s_b.clone(), b_bc.clone(), b_bd.clone()],
        vec![1.0; 16],
    )
    .unwrap();
    let t_c = IdxTensor::from_dense(vec![b_bc.clone(), s_c.clone()], vec![1.0; 4]).unwrap();
    let t_d =
        IdxTensor::from_dense(vec![b_bd.clone(), s_d.clone(), b_be.clone()], vec![1.0; 8]).unwrap();
    let t_e = IdxTensor::from_dense(vec![b_be.clone(), s_e.clone()], vec![1.0; 4]).unwrap();

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

fn build_uniform_chain_state_and_identity_operator(
    length: usize,
    state_bond_dim: usize,
    mpo_bond_dim: usize,
) -> ChainStateAndIdentity {
    let site_indices: Vec<_> = (0..length).map(|_| make_index(2)).collect();
    let state_bonds: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| make_index(state_bond_dim))
        .collect();
    let mpo_bonds: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| make_index(mpo_bond_dim))
        .collect();

    let mut state_tensors = Vec::with_capacity(length);
    let mut mpo_tensors = Vec::with_capacity(length);
    let mut node_names = Vec::with_capacity(length);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for site in 0..length {
        let name = format!("site{site}");
        node_names.push(name.clone());

        let mut state_indices = Vec::new();
        if site > 0 {
            state_indices.push(state_bonds[site - 1].clone());
        }
        state_indices.push(site_indices[site].clone());
        if site + 1 < length {
            state_indices.push(state_bonds[site].clone());
        }
        let state_len = state_indices.iter().map(|idx| idx.dim()).product();
        state_tensors.push(IdxTensor::from_dense(state_indices, vec![1.0; state_len]).unwrap());

        let internal_input = make_index(2);
        let internal_output = make_index(2);
        let mut mpo_indices = Vec::new();
        if site > 0 {
            mpo_indices.push(mpo_bonds[site - 1].clone());
        }
        mpo_indices.push(internal_output.clone());
        mpo_indices.push(internal_input.clone());
        if site + 1 < length {
            mpo_indices.push(mpo_bonds[site].clone());
        }

        let mpo_len = mpo_indices.iter().map(|idx| idx.dim()).product();
        let mut mpo_data = vec![0.0; mpo_len];
        let left_dim = if site > 0 { mpo_bond_dim } else { 1 };
        let right_dim = if site + 1 < length { mpo_bond_dim } else { 1 };
        for left in 0..left_dim {
            for output in 0..2 {
                for input in 0..2 {
                    for right in 0..right_dim {
                        if output == input {
                            let offset = if site == 0 {
                                output + 2 * (input + 2 * right)
                            } else if site + 1 == length {
                                left + left_dim * (output + 2 * input)
                            } else {
                                left + left_dim * (output + 2 * (input + 2 * right))
                            };
                            mpo_data[offset] = 1.0 / mpo_bond_dim as f64;
                        }
                    }
                }
            }
        }
        mpo_tensors.push(IdxTensor::from_dense(mpo_indices, mpo_data).unwrap());

        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: site_indices[site].clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: site_indices[site].clone(),
                internal_index: internal_output,
            },
        );
    }

    let state = TreeTN::from_tensors(state_tensors, node_names.clone()).unwrap();
    let mpo = TreeTN::from_tensors(mpo_tensors, node_names).unwrap();
    let sites = site_indices
        .into_iter()
        .enumerate()
        .map(|(i, idx)| (format!("site{i}"), idx))
        .collect();

    (
        state,
        LinearOperator::new(mpo, input_mapping, output_mapping),
        sites,
    )
}

fn assert_identity_application(
    state: &TreeTN<IdxTensor, String>,
    operator: &LinearOperator<IdxTensor, String>,
) {
    use crate::operator::apply_linear_operator;

    let result = apply_linear_operator(operator, state, ApplyOptions::default()).unwrap();
    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn test_apply_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = ApplyOptions::<usize>::zipup()
        .with_max_bond_dim(50)
        .with_svd_policy(policy);
    assert_eq!(opts.method, ContractionMethod::Zipup);
    assert_eq!(opts.max_bond_dim, Some(50));
    assert_eq!(opts.svd_policy, Some(policy));
    assert_eq!(opts.qr_rtol, None);

    let fit = ApplyOptions::<usize>::fit().with_nfullsweeps(3);
    assert_eq!(fit.method, ContractionMethod::Fit);
    assert_eq!(fit.nfullsweeps, 3);

    let naive = ApplyOptions::<usize>::naive();
    assert_eq!(naive.method, ContractionMethod::Naive);
}

#[test]
fn apply_options_explicit_center_is_honored_on_a_chain() {
    // Non-`usize` String node names, and both the naive-local and contraction
    // dispatch paths, must canonicalize the result at the requested centre.
    use crate::operator::apply_linear_operator;

    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[2].clone()]);
    for (label, method) in [
        ("naive", ApplyOptions::naive()),
        ("zipup", ApplyOptions::zipup()),
    ] {
        let options = ApplyOptions {
            center: Some("site1".to_string()),
            ..method
        };
        let result = apply_linear_operator(&operator, &state, options).unwrap();
        assert!(
            result.canonical_region().contains("site1"),
            "{label}: expected the result centred on site1, got {:?}",
            result.canonical_region()
        );
        let distance = result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap();
        assert!(distance < 1.0e-10, "{label}: identity distance {distance}");
    }
}

#[test]
fn apply_options_default_center_matches_preferred_center() {
    use crate::operator::apply_linear_operator;

    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[2].clone()]);
    let preferred = state
        .preferred_contraction_center()
        .expect("chain has a preferred centre");
    assert_eq!(preferred, "site0".to_string());

    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert!(result.canonical_region().contains(&preferred));
}

#[test]
fn apply_options_rejects_a_center_that_is_not_a_state_node() {
    use crate::operator::apply_linear_operator;

    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[2].clone()]);
    let options = ApplyOptions {
        center: Some("missing".to_string()),
        ..ApplyOptions::default()
    };
    let err = apply_linear_operator(&operator, &state, options).unwrap_err();
    assert!(
        err.to_string().contains("is not a node of the state"),
        "{err}"
    );
}

#[test]
fn apply_linear_operator_src_preserves_a_two_site_identity() {
    let site0 = make_index(2);
    let site1 = make_index(2);
    let state_bond = make_index(2);
    let state = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![site0.clone(), state_bond.clone()],
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![state_bond, site1.clone()], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();
    let operator = build_bonded_identity_operator(&[
        ("site0".to_string(), site0),
        ("site1".to_string(), site1),
    ]);

    let result = apply_linear_operator(
        &operator,
        &state,
        ApplyOptions::src()
            .with_max_bond_dim(4)
            .with_src_options(SrcOptions::fixed().with_final_svd(false).with_seed(17)),
    )
    .unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1.0e-10
    );
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
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space).unwrap();

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
    assert_eq!(lin_op.num_external_indices(), 2);

    // Test replaceind
    let new_idx = make_index(2);
    let replaced = lin_op.replaceind(&true_s0, &new_idx).unwrap();
    assert!(replaced.get_input_mapping(&"N0".to_string()).is_some());

    let same_id_prime = true_s0.prime();
    let wrong_prime = lin_op.replaceind(&same_id_prime, &make_index(2));
    assert!(wrong_prime.is_err());
}

#[test]
fn test_linear_operator_replaceind_errors() {
    let operator = build_identity_operator(&[("site0".to_string(), make_index(2))]);

    let mismatch = operator.replaceind(&make_index(2), &make_index(3));
    assert!(mismatch.is_err());

    let unknown = operator.replaceind(&make_index(2), &make_index(2));
    assert!(unknown.is_err());
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
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space).unwrap();

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
fn test_arc_linear_operator_accessors_and_conversion() {
    let op = build_identity_operator(&[("site0".to_string(), make_index(2))]);
    let arc_op = ArcLinearOperator::from_linear_operator(op.clone());

    assert_eq!(arc_op.node_names(), op.node_names());
    assert!(arc_op.get_input_mapping(&"site0".to_string()).is_some());
    assert!(arc_op.get_output_mapping(&"site0".to_string()).is_some());
    assert_eq!(arc_op.input_mappings().len(), 1);
    assert_eq!(arc_op.output_mappings().len(), 1);

    let arc_clone = arc_op.clone();
    let op2 = arc_clone.into_linear_operator();
    assert_eq!(op2.mpo.node_count(), op.mpo.node_count());
}

#[test]
fn test_apply_linear_operator_full_coverage() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 2-site state
    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let b01 = make_index(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();

    // Create identity operator covering both sites
    let mut mpo = TreeTN::<IdxTensor, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data).unwrap();

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
fn naive_apply_preserves_product_link_space_for_redundant_mpo_bond() {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let state_bond = make_index(2);
    let state = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), state_bond.clone()],
                vec![1.0, 3.0, 2.0, 5.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![state_bond.clone(), s1.clone()],
                vec![7.0, 11.0, 13.0, 17.0],
            )
            .unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();
    let operator = build_redundant_bonded_identity_operator(
        &[
            ("site0".to_string(), s0.clone()),
            ("site1".to_string(), s1.clone()),
        ],
        3,
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    let edge = result
        .edge_between(&"site0".to_string(), &"site1".to_string())
        .unwrap();
    assert_eq!(result.bond_index(edge).unwrap().dim(), 2 * 3);

    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn naive_apply_full_product_identity_embeds_on_state_topology() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&sites);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), state.node_count());
    assert_eq!(result.edge_count(), state.edge_count());

    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn naive_apply_noncontiguous_bonded_identity_uses_compact_bridge_delta() {
    let (state, sites) = build_chain_state();
    let operator =
        build_redundant_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()], 2);

    let (extended, _) = extend_operator_to_full_space(&operator, &state).unwrap();
    let middle = extended.mpo.node_index(&sites[1].0).unwrap();
    let middle_tensor = extended.mpo.tensor(middle).unwrap();

    assert_eq!(
        middle_tensor.storage().unwrap().storage_kind(),
        StorageKind::Structured
    );
    assert_eq!(middle_tensor.storage().unwrap().payload_dims(), &[2, 2]);
    assert_eq!(
        middle_tensor.storage().unwrap().axis_classes(),
        &[0, 0, 1, 1]
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn naive_apply_long_identity_chain_keeps_local_bonds_bounded() {
    // Keep this large enough that dense materialization would be infeasible,
    // while local exact apply remains linear in the chain length.
    let length = 24;
    let state_bond_dim = 2;
    let mpo_bond_dim = 1;
    let (state, operator, sites) =
        build_uniform_chain_state_and_identity_operator(length, state_bond_dim, mpo_bond_dim);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), state.node_count());
    assert_eq!(result.edge_count(), state.edge_count());
    let mut edges: Vec<_> = result.site_index_network().edges().collect();
    edges.sort();
    for (left, right) in edges {
        let edge = result.edge_between(&left, &right).unwrap();
        assert!(
            result.bond_index(edge).unwrap().dim() <= state_bond_dim * mpo_bond_dim,
            "unexpected bond dimension on edge {left:?}-{right:?}"
        );
    }

    let site_indices: Vec<_> = sites.into_iter().map(|(_, idx)| idx).collect();
    let sample_points = [
        vec![0; length],
        vec![1; length],
        (0..length).map(|i| i % 2).collect::<Vec<_>>(),
    ];
    let mut values = Vec::with_capacity(length * sample_points.len());
    for point in &sample_points {
        values.extend(point.iter().copied());
    }
    let shape = [length, sample_points.len()];
    let value_ref = ColMajorArrayRef::new(&values, &shape).unwrap();
    let state_values = state.evaluate(&site_indices, value_ref).unwrap();
    let result_values = result.evaluate(&site_indices, value_ref).unwrap();

    for (state_value, result_value) in state_values.iter().zip(result_values.iter()) {
        assert!((state_value.real() - result_value.real()).abs() < 1e-10);
    }
}

#[test]
fn test_apply_linear_operator_partial() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 3-site state
    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        IdxTensor::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 = IdxTensor::from_dense(vec![b12.clone(), s2.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    let n2 = state.add_tensor("site2".to_string(), t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();

    // Create operator covering only site0 and site1 (partial)
    let mut mpo = TreeTN::<IdxTensor, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data).unwrap();

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
fn test_apply_linear_operator_partial_preserves_site_index_set() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        IdxTensor::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 = IdxTensor::from_dense(vec![b12.clone(), s2.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    let n2 = state.add_tensor("site2".to_string(), t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();

    let mut mpo = TreeTN::<IdxTensor, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data).unwrap();

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

    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    let (result_site_indices, _) = result.all_site_indices().unwrap();
    let result_site_ids: HashSet<_> = result_site_indices.iter().map(|idx| *idx.id()).collect();
    let expected_site_ids: HashSet<_> = [s0, s1, s2].iter().map(|idx| *idx.id()).collect();

    assert_eq!(result_site_ids, expected_site_ids);
    assert_eq!(result_site_indices.len(), expected_site_ids.len());
}

#[test]
fn test_apply_linear_operator_partial_shift_factorized_rooted_state_intermediate_indices() {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let spectator = make_index(2);
    let dense = IdxTensor::from_dense(
        vec![s0.clone(), s1.clone(), spectator.clone()],
        vec![10.0, 30.0, 20.0, 40.0, 11.0, 31.0, 21.0, 41.0],
    )
    .unwrap();

    let mut nodes = HashMap::new();
    nodes.insert(0usize, vec![s0.clone()]);
    nodes.insert(1usize, vec![s1.clone()]);
    nodes.insert(2usize, vec![spectator.clone()]);
    let topology = crate::TreeTopology::new(nodes, vec![(0usize, 1usize), (1usize, 2usize)]);
    let state = crate::factorize_tensor_to_treetn(&dense, &topology, &2usize).unwrap();
    let mut state_edges = state.site_index_network().edges().collect::<Vec<_>>();
    state_edges.sort();
    assert_eq!(state_edges, vec![(0usize, 1usize), (1usize, 2usize)]);

    let mut mpo = TreeTN::<IdxTensor, usize>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data).unwrap();

    let n0_mpo = mpo.add_tensor(0usize, t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor(1usize, t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    input_mapping.insert(
        0usize,
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        1usize,
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        0usize,
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        1usize,
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    let (full_operator, _) = extend_operator_to_full_space(&operator, &state).unwrap();
    let mut output_keys: Vec<_> = full_operator.output_mappings().keys().cloned().collect();
    output_keys.sort();
    assert_eq!(output_keys, vec![0usize, 1usize, 2usize]);
    let transformed_state = transform_state_to_input(&full_operator, &state).unwrap();
    let contracted_tensor = transformed_state
        .contract_naive(full_operator.mpo())
        .unwrap();

    let contracted_ids: HashSet<_> = contracted_tensor
        .external_indices()
        .iter()
        .map(|idx| *idx.id())
        .collect();
    let expected_ids: HashSet<_> = [
        *full_operator
            .get_output_mapping(&0)
            .unwrap()
            .internal_index
            .id(),
        *full_operator
            .get_output_mapping(&1)
            .unwrap()
            .internal_index
            .id(),
        *full_operator
            .get_output_mapping(&2)
            .unwrap()
            .internal_index
            .id(),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        contracted_ids,
        expected_ids,
        "input ids {:?}, output ids {:?}, contracted ids {:?}",
        [
            *full_operator
                .get_input_mapping(&0)
                .unwrap()
                .internal_index
                .id(),
            *full_operator
                .get_input_mapping(&1)
                .unwrap()
                .internal_index
                .id(),
        ],
        [
            *full_operator
                .get_output_mapping(&0)
                .unwrap()
                .internal_index
                .id(),
            *full_operator
                .get_output_mapping(&1)
                .unwrap()
                .internal_index
                .id(),
        ],
        contracted_tensor
            .external_indices()
            .iter()
            .map(|idx| *idx.id())
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        contracted_tensor.external_indices().len(),
        expected_ids.len()
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let (result_site_indices, _) = result.all_site_indices().unwrap();
    let result_site_ids: HashSet<_> = result_site_indices.iter().map(|idx| *idx.id()).collect();
    let expected_true_ids: HashSet<_> = [s0, s1, spectator].iter().map(|idx| *idx.id()).collect();
    assert_eq!(result_site_ids, expected_true_ids);
    assert_eq!(result_site_indices.len(), expected_true_ids.len());
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
fn test_apply_linear_operator_non_contiguous_chain_with_bonded_operator() {
    let (state, sites) = build_chain_state();
    let operator = build_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()]);

    let result =
        crate::operator::apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), state.node_count());
}

#[test]
fn test_apply_linear_operator_non_contiguous_tree_with_bonded_operator() {
    let (state, sites) = build_tree_state();
    let operator = build_bonded_identity_operator(&[sites[0].clone(), sites[4].clone()]);

    let result =
        crate::operator::apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), state.node_count());
}

#[test]
fn test_apply_linear_operator_error_cases() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 2-site state
    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let b01 = make_index(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();

    // Create operator with extra node not in state (should error)
    let mut mpo = TreeTN::<IdxTensor, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s2_in = make_index(2); // site2 doesn't exist in state
    let s2_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t2_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s2_out.clone(), s2_in.clone()], id_data).unwrap();

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
fn test_extend_operator_to_full_space_rejects_missing_state_node() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[("ghost".to_string(), sites[0].1.clone())]);

    let err = extend_operator_to_full_space(&operator, &state).unwrap_err();
    assert!(err.to_string().contains("missing from the state network"));
}

#[test]
fn test_extend_operator_to_full_space_uses_unchecked_branch_for_single_node_operator() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone()]);

    let (extended, _) = extend_operator_to_full_space(&operator, &state).unwrap();
    assert_eq!(extended.mpo.node_count(), state.node_count());
    assert_eq!(extended.mpo.edge_count(), state.edge_count());
}

#[test]
fn test_compose_operator_along_state_paths_scalar_identity_gap() {
    let mut state_network: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let a = make_index(2);
    state_network
        .add_node(
            "A".to_string(),
            [a.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    state_network
        .add_node("B".to_string(), HashSet::new())
        .unwrap();
    state_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let operator = build_identity_operator(&[("A".to_string(), a.clone())]);
    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert("B".to_string(), Vec::new());

    let composed = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap();

    assert_eq!(composed.mpo.node_count(), 2);
    assert_eq!(composed.mpo.edge_count(), 1);
}

#[test]
fn test_compose_operator_along_state_paths_missing_gap_indices() {
    let (state, sites) = build_chain_state();
    let state_network = state.site_index_network().clone();
    let operator = build_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()]);

    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert("site0".to_string(), Vec::new());
    gap_site_indices.insert("site2".to_string(), Vec::new());

    let err = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("missing gap indices"));
}

#[test]
fn test_compose_operator_along_state_paths_missing_state_node() {
    let mut state_network: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let a = make_index(2);
    state_network
        .add_node(
            "A".to_string(),
            [a.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();

    let ghost = make_index(2);
    let mut mpo = TreeTN::new();
    let a_in = make_index(2);
    let a_out = make_index(2);
    let g_in = make_index(2);
    let g_out = make_index(2);
    let bond = make_index(1);

    let t_a = IdxTensor::from_dense(
        vec![a_out.clone(), a_in.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let t_g = IdxTensor::from_dense(
        vec![bond.clone(), g_out.clone(), g_in.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let n_a = mpo.add_tensor("A".to_string(), t_a).unwrap();
    let n_g = mpo.add_tensor("Ghost".to_string(), t_g).unwrap();
    mpo.connect(n_a, &bond, n_g, &bond).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    input_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: a.clone(),
            internal_index: a_in.clone(),
        },
    );
    input_mapping.insert(
        "Ghost".to_string(),
        IndexMapping {
            true_index: ghost.clone(),
            internal_index: g_in.clone(),
        },
    );
    output_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: a.clone(),
            internal_index: a_out.clone(),
        },
    );
    output_mapping.insert(
        "Ghost".to_string(),
        IndexMapping {
            true_index: ghost.clone(),
            internal_index: g_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);
    let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

    let err = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("missing state node"));
}

#[test]
fn test_compose_operator_along_state_paths_no_path_between_operator_nodes() {
    let mut state_network: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let a = make_index(2);
    let c = make_index(2);
    state_network
        .add_node(
            "A".to_string(),
            [a.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    state_network
        .add_node(
            "C".to_string(),
            [c.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();

    let operator = build_bonded_identity_operator(&[
        ("A".to_string(), a.clone()),
        ("C".to_string(), c.clone()),
    ]);
    let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

    let err = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("no path between"));
}

#[test]
fn test_linear_operator_replace_indices() {
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
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space).unwrap();

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

    // Test replace_indices
    let new_idx1 = make_index(2);
    let new_idx2 = make_index(2);
    let replaced = lin_op
        .replace_indices(
            std::slice::from_ref(&true_s0),
            std::slice::from_ref(&new_idx1),
        )
        .unwrap();
    assert_eq!(
        replaced
            .get_input_mapping(&"N0".to_string())
            .unwrap()
            .true_index,
        new_idx1
    );
    assert_eq!(
        replaced
            .get_output_mapping(&"N0".to_string())
            .unwrap()
            .true_index,
        new_idx1
    );

    // Replacing the same old index twice is ambiguous now that replaceind
    // updates all matching input/output mappings.
    let new_idx3 = make_index(2);
    let duplicate =
        lin_op.replace_indices(&[true_s0.clone(), true_s0.clone()], &[new_idx2, new_idx3]);
    assert!(duplicate.is_err());
    assert!(duplicate
        .unwrap_err()
        .to_string()
        .contains("Duplicate old index"));

    // Test replace_indices error case (length mismatch)
    let result = lin_op.replace_indices(
        std::slice::from_ref(&true_s0),
        &[new_idx1.clone(), new_idx1],
    );
    assert!(result.is_err());
}

/// Chain `site0 - site1 - site2 - site3` where `site1` stores two site legs as
/// `[s1b, bond01, s1a, bond12]`. Returns the input tensors (for repeated
/// `from_tensors` rebuilds), node names, all site indices per node, and the
/// two-site node's legs in tensor order.
#[allow(clippy::type_complexity)]
fn gap_order_chain() -> (
    Vec<IdxTensor>,
    Vec<String>,
    Vec<(String, DynIndex)>,
    Vec<DynIndex>,
) {
    let s0 = make_index(2);
    let s1a = make_index(2);
    let s1b = make_index(2);
    let s2 = make_index(2);
    let s3 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);
    let b23 = make_index(2);
    let fill = |len: usize, salt: usize| -> Vec<f64> {
        (0..len)
            .map(|i| ((i * 5 + salt * 11) as f64 * 0.413).sin())
            .collect()
    };
    let tensors = vec![
        IdxTensor::from_dense(vec![s0.clone(), b01.clone()], fill(4, 0)).unwrap(),
        IdxTensor::from_dense(
            vec![s1b.clone(), b01.clone(), s1a.clone(), b12.clone()],
            fill(16, 1),
        )
        .unwrap(),
        IdxTensor::from_dense(vec![b12.clone(), s2.clone(), b23.clone()], fill(8, 2)).unwrap(),
        IdxTensor::from_dense(vec![b23.clone(), s3.clone()], fill(4, 3)).unwrap(),
    ];
    let names: Vec<String> = (0..4).map(|i| format!("site{i}")).collect();
    let sites = vec![(names[2].clone(), s2), (names[3].clone(), s3)];
    (tensors, names, sites, vec![s1b, s1a])
}

#[test]
fn compose_with_gaps_uses_a_deterministic_node_order() {
    let (tensors, names, op_sites, _) = gap_order_chain();
    for _ in 0..64 {
        let state =
            TreeTN::<IdxTensor, String>::from_tensors(tensors.clone(), names.clone()).unwrap();
        let operator = build_bonded_identity_operator(&op_sites);
        // Identity gaps at site0 and site1 (site1 carries two site legs).
        let mut gaps: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
        for gap in ["site0", "site1"] {
            let pairs = state
                .node_site_indices(&gap.to_string())
                .unwrap()
                .iter()
                .map(|index| (index.sim(), index.sim()))
                .collect();
            gaps.insert(gap.to_string(), pairs);
        }
        let composed =
            compose_exclusive_linear_operators(state.site_index_network(), &[&operator], &gaps)
                .unwrap();
        // Operator nodes sorted by name, then gap nodes sorted by name.
        assert_eq!(
            composed.mpo().node_names(),
            vec!["site2", "site3", "site0", "site1"]
        );
    }
}

#[test]
fn apply_with_identity_gaps_keeps_the_state_site_leg_order() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    let (tensors, names, op_sites, site1_legs) = gap_order_chain();
    let mut reference: Option<Vec<u64>> = None;
    for _ in 0..64 {
        let state =
            TreeTN::<IdxTensor, String>::from_tensors(tensors.clone(), names.clone()).unwrap();
        let operator = build_bonded_identity_operator(&op_sites);
        let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
        // The identity gap at site1 keeps the state's own leg order.
        assert_eq!(
            result.node_site_indices(&"site1".to_string()).unwrap(),
            site1_legs
        );
        let dense = result.to_dense().unwrap();
        assert_eq!(
            dense.external_indices(),
            state.to_dense().unwrap().external_indices()
        );
        let bits: Vec<u64> = dense
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        match &reference {
            None => reference = Some(bits),
            Some(reference) => assert_eq!(&bits, reference),
        }
    }
}

// ---------------------------------------------------------------------------
// Issue #1186: a partial apply must not factorize the nodes its operator does
// not act on.
// ---------------------------------------------------------------------------

/// Deterministic, non-constant tensor values (an LCG).
struct CombValues(u64);

impl CombValues {
    fn take(&mut self, count: usize) -> Vec<f64> {
        (0..count)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 0.5
            })
            .collect()
    }
}

/// A comb: the degree-three scalar centre `hub`, the branch `a0-a1-a2` an
/// operator can act on, and the branches `b0-b1` and `d0-d1` no operator
/// touches. State bonds have dimension three and site legs dimension two, so a
/// leg's role is visible in its dimension.
fn build_comb_state() -> (TreeTN<IdxTensor, String>, Vec<(String, DynIndex)>) {
    let mut values = CombValues(0x5eed_1186);
    let s_hub = make_index(1);
    let s_a: Vec<DynIndex> = (0..3).map(|_| make_index(2)).collect();
    let s_b: Vec<DynIndex> = (0..2).map(|_| make_index(2)).collect();
    let s_d: Vec<DynIndex> = (0..2).map(|_| make_index(2)).collect();
    let c_a: Vec<DynIndex> = (0..3).map(|_| make_index(3)).collect();
    let c_b: Vec<DynIndex> = (0..2).map(|_| make_index(3)).collect();
    let c_d: Vec<DynIndex> = (0..2).map(|_| make_index(3)).collect();

    let mut tensors = vec![IdxTensor::from_dense(
        vec![
            s_hub.clone(),
            c_a[0].clone(),
            c_b[0].clone(),
            c_d[0].clone(),
        ],
        values.take(27),
    )
    .unwrap()];
    let mut names = vec!["hub".to_string()];
    let mut sites = vec![("hub".to_string(), s_hub)];

    for (prefix, bonds, branch_sites) in [("a", &c_a, &s_a), ("b", &c_b, &s_b), ("d", &c_d, &s_d)] {
        for (i, site) in branch_sites.iter().enumerate() {
            let mut legs = vec![bonds[i].clone(), site.clone()];
            let mut count = 3 * 2;
            if i + 1 < bonds.len() {
                legs.push(bonds[i + 1].clone());
                count *= 3;
            }
            tensors.push(IdxTensor::from_dense(legs, values.take(count)).unwrap());
            names.push(format!("{prefix}{i}"));
            sites.push((format!("{prefix}{i}"), site.clone()));
        }
    }

    let state = TreeTN::from_tensors(tensors, names).unwrap();
    (state, sites)
}

/// An operator with bond dimension two on the branch `a0-a1-a2` only.
fn build_branch_operator(sites: &[(String, DynIndex)]) -> LinearOperator<IdxTensor, String> {
    let true_of = |name: &str| {
        sites
            .iter()
            .find(|(node, _)| node == name)
            .unwrap()
            .1
            .clone()
    };
    let inputs: Vec<DynIndex> = (0..3).map(|_| make_index(2)).collect();
    let outputs: Vec<DynIndex> = (0..3).map(|_| make_index(2)).collect();
    let bond_01 = make_index(2);
    let bond_12 = make_index(2);

    let t0 = IdxTensor::from_dense(
        vec![inputs[0].clone(), outputs[0].clone(), bond_01.clone()],
        vec![1.0, 0.0, 0.0, 0.3, 0.0, 1.0, 1.0, 0.4],
    )
    .unwrap();
    let mut middle = vec![0.0; 16];
    for bond in 0..2 {
        for input in 0..2 {
            for output in 0..2 {
                middle[((bond * 2 + input) * 2 + output) * 2 + bond] =
                    if input == output { 1.0 } else { 0.7 };
            }
        }
    }
    let t1 = IdxTensor::from_dense(
        vec![
            bond_01.clone(),
            inputs[1].clone(),
            outputs[1].clone(),
            bond_12.clone(),
        ],
        middle,
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![bond_12.clone(), inputs[2].clone(), outputs[2].clone()],
        vec![1.0, 0.0, 0.5, 0.0, 1.0, 2.0, 0.0, 1.0],
    )
    .unwrap();

    let mpo = TreeTN::from_tensors(
        vec![t0, t1, t2],
        vec!["a0".to_string(), "a1".to_string(), "a2".to_string()],
    )
    .unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    for i in 0..3 {
        let name = format!("a{i}");
        let true_index = true_of(&name);
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: inputs[i].clone(),
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index,
                internal_index: outputs[i].clone(),
            },
        );
    }

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn tensor_bits(tensor: &IdxTensor) -> Vec<u64> {
    tensor
        .to_vec::<f64>()
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect()
}

/// Assert that `name` came out of a zip-up as the state's own tensor, bit for
/// bit: the site indices keep their identity, the bond legs are the state's in
/// the same order with fresh IDs, and every amplitude is unchanged.
fn assert_node_passed_through(
    result: &TreeTN<IdxTensor, String>,
    state: &TreeTN<IdxTensor, String>,
    name: &str,
) {
    let node = name.to_string();
    let output = result.tensor(result.node_index(&node).unwrap()).unwrap();
    let input = state.tensor(state.node_index(&node).unwrap()).unwrap();
    let site_indices = state.node_site_indices(&node).unwrap();
    assert_eq!(
        result.node_site_indices(&node).unwrap(),
        site_indices,
        "{name} keeps the state's site indices"
    );
    let bonds_of = |tensor: &IdxTensor| -> Vec<DynIndex> {
        tensor
            .external_indices()
            .into_iter()
            .filter(|index| !site_indices.contains(index))
            .collect()
    };
    let output_bonds = bonds_of(output);
    let input_bonds = bonds_of(input);
    assert_eq!(output_bonds.len(), input_bonds.len(), "{name} leg count");
    // The pass-through keeps the bond legs in the state's order; only their IDs
    // are fresh (the contraction simulates internal indices on both operands).
    let mut realigned = output.clone();
    for (fresh, original) in output_bonds.iter().zip(input_bonds.iter()) {
        realigned = realigned.replaceind(fresh, original).unwrap();
    }
    // The identity factor's delta decides the leg order of what it emits (its
    // output legs first), so compare the leg set and the amplitudes up to that
    // permutation.
    let ordered = realigned
        .permuteinds(&input.external_indices())
        .unwrap_or_else(|error| panic!("{name} does not carry the state's legs: {error}"));
    assert_eq!(
        tensor_bits(&ordered),
        tensor_bits(input),
        "{name} tensor data"
    );
}

/// Largest absolute difference between two amplitude sequences.
///
/// `IdxTensor::sub` plus `maxabs` cannot express this comparison here: an apply
/// result keeps the identity extension's dimension-one legs and the exact local
/// (`naive`) reference does not, so the two dense tensors' leg counts differ and
/// `sub` rejects them. Every node of these states carries exactly one site
/// index, so `to_dense` orders both results' legs by node name and their
/// amplitude sequences line up, and the dimension-one legs carry no data.
///
/// The finiteness guard keeps a NaN or an infinity out of `f64::max`, which
/// would silently ignore it.
fn max_finite_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "amplitude counts");
    for (index, (new, reference)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (new - reference).is_finite(),
            "amplitude {index} is not comparable: {new:e} vs {reference:e}"
        );
    }
    left.iter()
        .zip(right.iter())
        .map(|(new, reference)| (new - reference).abs())
        .fold(0.0, f64::max)
}

fn dense_deviation(left: &TreeTN<IdxTensor, String>, right: &TreeTN<IdxTensor, String>) -> f64 {
    let left = left.to_dense().unwrap().to_vec::<f64>().unwrap();
    let right = right.to_dense().unwrap().to_vec::<f64>().unwrap();
    max_finite_abs_difference(&left, &right)
}

#[test]
#[should_panic(expected = "is not comparable")]
fn max_finite_abs_difference_rejects_a_non_finite_difference() {
    assert_eq!(
        max_finite_abs_difference(&[0.0, 1.0], &[0.0, f64::NAN]),
        0.0
    );
}

#[test]
fn apply_zipup_skips_the_nodes_its_operator_does_not_touch() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;
    use crate::treetn::contraction::zipup_identity_node_skips;

    let (state, sites) = build_comb_state();
    let state = state
        .canonicalize(["hub".to_string()], Default::default())
        .unwrap();
    let operator = build_branch_operator(&sites);

    let skipped_before = zipup_identity_node_skips();
    let result = apply_linear_operator(
        &operator,
        &state,
        ApplyOptions {
            center: Some("hub".to_string()),
            // The skip is an opt-in for exactness: a zero threshold means the
            // factorizations it replaces cannot cut anything. An absent policy
            // does not qualify (it resolves to the default cutoff).
            svd_policy: Some(SvdTruncationPolicy::new(0.0)),
            ..ApplyOptions::zipup()
        },
    )
    .unwrap();

    // The comb has eight nodes; the four in the two untouched branches
    // (`b0`, `b1`, `d0`, `d1`) are the four of the seven non-hub nodes that
    // nothing touches. `hub` receives the operator branch's remainder, so it is
    // still contracted and factorized.
    let skipped = zipup_identity_node_skips() - skipped_before;
    assert_eq!(skipped, 4, "untouched branch nodes skipped");
    result.validate_ortho_consistency().unwrap();
    assert_eq!(
        result.canonical_region(),
        &["hub".to_string()].into_iter().collect()
    );

    for name in ["b0", "b1", "d0", "d1"] {
        assert_node_passed_through(&result, &state, name);
    }

    for (left, right) in [("hub", "b0"), ("b0", "b1"), ("hub", "d0"), ("d0", "d1")] {
        let output_edge = result
            .edge_between(&left.to_string(), &right.to_string())
            .unwrap();
        let input_edge = state
            .edge_between(&left.to_string(), &right.to_string())
            .unwrap();
        assert_eq!(
            result.bond_index(output_edge).unwrap().dim(),
            state.bond_index(input_edge).unwrap().dim(),
            "the untouched bond {left}-{right} keeps its dimension"
        );
    }

    // The untouched region is carried over unchanged, not approximated: the
    // only tolerance is the operator branch's own factorization.
    let exact = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let deviation = dense_deviation(&result, &exact);
    assert!(
        deviation < 1.0e-12,
        "the zip-up result deviates from the exact local apply by {deviation:e}"
    );
}

/// A three-site chain whose two leading sites carry an operator, so the chain's
/// head is the only node the operator does not touch.
fn build_chain_state_and_operator() -> (TreeTN<IdxTensor, String>, LinearOperator<IdxTensor, String>)
{
    let mut values = CombValues(0xc4a1_1186);
    let site_0 = make_index(2);
    let site_1 = make_index(2);
    let site_2 = make_index(2);
    let bond_01 = make_index(3);
    let bond_12 = make_index(3);

    let state = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![site_0.clone(), bond_01.clone()], values.take(6)).unwrap(),
            IdxTensor::from_dense(
                vec![bond_01.clone(), site_1.clone(), bond_12.clone()],
                values.take(18),
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond_12.clone(), site_2.clone()], values.take(6)).unwrap(),
        ],
        vec![
            "site0".to_string(),
            "site1".to_string(),
            "site2".to_string(),
        ],
    )
    .unwrap();

    let inputs: Vec<DynIndex> = (0..2).map(|_| make_index(2)).collect();
    let outputs: Vec<DynIndex> = (0..2).map(|_| make_index(2)).collect();
    let mpo_bond = make_index(2);
    let mpo = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![inputs[0].clone(), outputs[0].clone(), mpo_bond.clone()],
                vec![0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 1.5, 0.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![mpo_bond.clone(), inputs[1].clone(), outputs[1].clone()],
                vec![1.0, 0.0, 0.25, 0.0, 1.0, -0.5, 0.0, 1.0],
            )
            .unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    for (i, true_index) in [(0, site_0), (1, site_1)] {
        let name = format!("site{i}");
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: inputs[i].clone(),
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index,
                internal_index: outputs[i].clone(),
            },
        );
    }

    (
        state,
        LinearOperator::new(mpo, input_mapping, output_mapping),
    )
}

#[test]
fn apply_zipup_leaves_a_chain_partial_apply_on_the_chain_schedule() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;
    use crate::treetn::contraction::zipup_identity_node_skips;

    let (state, operator) = build_chain_state_and_operator();

    let skipped_before = zipup_identity_node_skips();
    let result = apply_linear_operator(&operator, &state, ApplyOptions::zipup()).unwrap();
    // The chain schedule hands one remainder per edge to its successor, so it
    // cannot emit a node without one; a chain-shaped state keeps that schedule,
    // which conditions both operands and truncates at the centre.
    assert_eq!(
        zipup_identity_node_skips() - skipped_before,
        0,
        "a chain-shaped state keeps the chain schedule"
    );

    let exact = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let deviation = dense_deviation(&result, &exact);
    assert!(
        deviation < 1.0e-12,
        "the zipup result deviates from the exact local apply by {deviation:e}"
    );
}

/// The apply pipeline with the identity-node skip switched off: the reference
/// for "the result must not change when the untouched nodes are skipped".
fn apply_without_identity_skip(
    operator: &LinearOperator<IdxTensor, String>,
    state: &TreeTN<IdxTensor, String>,
    center: &str,
    options: &ApplyOptions<String>,
) -> TreeTN<IdxTensor, String> {
    let (full_operator, _) = extend_operator_to_full_space(operator, state).unwrap();
    let transformed = transform_state_to_input(&full_operator, state).unwrap();
    let contracted = transformed
        .contract_zipup_skipping_identity_nodes(
            full_operator.mpo(),
            &center.to_string(),
            options.svd_policy,
            options.max_bond_dim,
            &IdentityNodes::new(),
        )
        .unwrap();
    transform_output_to_true(&full_operator, contracted).unwrap()
}

/// The reviewer's four-node counterexample (#1197), kept as a fence.
///
/// Nodes `a` (site dim 1), `x` (1), `b` (2), `r` (2); bonds `a-x = 1`,
/// `b-x = 2`, `x-r = 2`. The untouched leaf carries `diag(1, 1e10)` and the hub
/// `diag(1, 1e-14)`, so the state's exact amplitudes on `b`'s site are `(1, 1e-4)`.
/// The network is deliberately **not** canonicalized: a canonical form is what
/// hid this from the existing fixtures.
///
/// If the node is passed through, its weights never reach `x`, and `x`'s cutoff
/// (`rtol = 1e-6`) drops the direction carrying `1e-4`: the result becomes
/// `(1, 0)`, an error of `1e-4`. This test asserts `(1, 1e-4)`, so widening the
/// skip back over a truncating policy fails here.
fn build_four_node_star() -> (TreeTN<IdxTensor, String>, LinearOperator<IdxTensor, String>) {
    let s_a = make_index(1);
    let s_x = make_index(1);
    let s_b = make_index(2);
    let s_r = make_index(2);
    let bond_ax = make_index(1);
    let bond_bx = make_index(2);
    let bond_xr = make_index(2);

    // Amplitude = a * x * b * r with `x` and `b` diagonal over their bond pair,
    // so the state reduces to a diagonal in `(s_b, s_r)`: 1 and 1e10 * 1e-14.
    let t_a = IdxTensor::from_dense(vec![s_a.clone(), bond_ax.clone()], vec![1.0]).unwrap();
    // Column-major over [site, bond_ax, bond_bx, bond_xr] with dims
    // [1, 1, 2, 2]: the flat position of (0, 0, i, j) is `i + 2 * j`, so
    // `diag(1, 1e-14)` is this.
    let t_x = IdxTensor::from_dense(
        vec![
            s_x.clone(),
            bond_ax.clone(),
            bond_bx.clone(),
            bond_xr.clone(),
        ],
        vec![1.0, 0.0, 0.0, 1.0e-14],
    )
    .unwrap();
    // Column-major over [site, bond_bx] with dims [2, 2]: `diag(1, 1e10)`.
    let t_b = IdxTensor::from_dense(
        vec![s_b.clone(), bond_bx.clone()],
        vec![1.0, 0.0, 0.0, 1.0e10],
    )
    .unwrap();
    let t_r = IdxTensor::from_dense(vec![bond_xr.clone(), s_r.clone()], vec![1.0, 0.0, 0.0, 1.0])
        .unwrap();

    let state = TreeTN::from_tensors(
        vec![t_a, t_x, t_b, t_r],
        vec![
            "a".to_string(),
            "x".to_string(),
            "b".to_string(),
            "r".to_string(),
        ],
    )
    .unwrap();

    // Identity on `a` only: a single-node MPO with no bonds.
    let input_a = make_index(1);
    let output_a = make_index(1);
    let mpo = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![input_a.clone(), output_a.clone()], vec![1.0]).unwrap()],
        vec!["a".to_string()],
    )
    .unwrap();
    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "a".to_string(),
        IndexMapping {
            true_index: s_a.clone(),
            internal_index: input_a,
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "a".to_string(),
        IndexMapping {
            true_index: s_a,
            internal_index: output_a,
        },
    );

    (
        state,
        LinearOperator::new(mpo, input_mapping, output_mapping),
    )
}

#[test]
fn apply_zipup_does_not_skip_a_node_that_is_not_isometric_toward_the_center() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;
    use crate::treetn::contraction::zipup_identity_node_skips;

    let (state, operator) = build_four_node_star();
    let options = ApplyOptions {
        center: Some("r".to_string()),
        svd_policy: Some(SvdTruncationPolicy::new(1.0e-12)),
        ..ApplyOptions::zipup()
    };

    let skipped_before = zipup_identity_node_skips();
    let result = apply_linear_operator(&operator, &state, options.clone()).unwrap();
    let skipped = zipup_identity_node_skips() - skipped_before;

    // The fence, asserted first because it is the consequence that matters: the
    // exact amplitudes are 1 and 1e-4. Pass `b` through and its weights never
    // reach `x`, whose cutoff then drops the direction carrying 1e-4 and returns
    // (1, 0) — a deviation of 1e-4, which this assertion catches before anyone
    // looks at a skip counter. Sorted-abs so the check does not depend on the
    // dense leg order; the two zeros are structural.
    let mut dense = result.to_dense().unwrap().to_vec::<f64>().unwrap();
    assert_eq!(dense.len(), 4, "the star's dense size");
    dense.iter_mut().for_each(|value| *value = value.abs());
    dense.sort_by(f64::total_cmp);
    let expected = [0.0, 0.0, 1.0e-4, 1.0];
    for (got, want) in dense.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() <= 1.0e-12 * want.max(1.0),
            "amplitude {got:e} against the exact answer {want:e} — this input loses the \
             dominant amplitude if the skip is enabled over a non-zero threshold"
        );
    }

    // A non-zero threshold is not an exactness opt-in, so nothing is skipped even
    // though `b` is a zip-up leaf the operator does not touch.
    assert_eq!(
        skipped, 0,
        "a non-zero threshold must not enable the pass-through"
    );

    let exact = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let deviation = dense_deviation(&result, &exact);
    assert!(
        deviation <= 1.0e-12,
        "the truncated result deviates from the exact local apply by {deviation:e}"
    );
    let reference = apply_without_identity_skip(&operator, &state, "r", &options);
    let deviation_to_reference = dense_deviation(&result, &reference);
    assert!(
        deviation_to_reference <= 1.0e-12,
        "the truncated result changed against the same apply with the skip off by \
         {deviation_to_reference:e}"
    );
}

/// The pass-through stands in for a factorization, so it is allowed only where
/// the caller has asked for that factorization to be exact. An absent policy is
/// not such a request: it resolves to the process-wide default cutoff.
#[test]
fn apply_zipup_never_skips_without_an_explicit_exact_policy() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;
    use crate::treetn::contraction::zipup_identity_node_skips;

    let (state, operator) = build_four_node_star();

    // Default path: no policy, no cap. Nothing may be skipped, so a caller who
    // never opted in keeps exactly the numerics it had before #1197.
    let default_options = ApplyOptions {
        center: Some("r".to_string()),
        ..ApplyOptions::zipup()
    };
    let skipped_before = zipup_identity_node_skips();
    let default_result = apply_linear_operator(&operator, &state, default_options.clone()).unwrap();
    assert_eq!(
        zipup_identity_node_skips() - skipped_before,
        0,
        "an absent policy is not an exactness opt-in"
    );

    // An explicit zero threshold is the same request for exactness, and buys the
    // skip. Since #1285 the same request factorizes every other node by
    // untruncated QR, so this is the fence for the skip under the QR form.
    let exact_options = ApplyOptions {
        center: Some("r".to_string()),
        svd_policy: Some(SvdTruncationPolicy::new(0.0)),
        ..ApplyOptions::zipup()
    };
    let skipped_before = zipup_identity_node_skips();
    let exact_result = apply_linear_operator(&operator, &state, exact_options.clone()).unwrap();
    assert_eq!(
        zipup_identity_node_skips() - skipped_before,
        1,
        "a zero threshold opts the untouched leaf into the pass-through"
    );

    default_result.validate_ortho_consistency().unwrap();
    exact_result.validate_ortho_consistency().unwrap();
    assert!(exact_result.canonical_region().is_empty());
    assert_eq!(exact_result.canonical_form(), None);

    // Both are exact against the naive apply, and the opted-in path is not less
    // accurate: skipping a node whose factorization cannot cut anything removes a
    // round trip instead of adding error.
    let naive = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let default_deviation = dense_deviation(&default_result, &naive);
    let exact_deviation = dense_deviation(&exact_result, &naive);
    assert!(
        default_deviation <= 1.0e-15,
        "the default path deviates from the exact local apply by {default_deviation:e}"
    );
    assert!(
        exact_deviation <= 1.0e-15,
        "the opted-in pass-through deviates from the exact local apply by {exact_deviation:e}"
    );
    assert!(
        exact_deviation <= default_deviation,
        "the pass-through ({exact_deviation:e}) must not be less accurate than the \
         factorized path ({default_deviation:e})"
    );

    // Under the QR form the pass-through is still a pure relabeling: the same
    // exact request with the skip off factorizes `b` by QR as well and gives
    // the same tensor.
    let unskipped = apply_without_identity_skip(&operator, &state, "r", &exact_options);
    let skip_deviation = dense_deviation(&exact_result, &unskipped);
    assert!(
        skip_deviation <= 1.0e-15,
        "the pass-through changed the exact QR result by {skip_deviation:e}"
    );
}

/// Regression for issue #1227: a swap must be applied from one old-to-new map,
/// not pair by pair, which used to collapse both mappings onto `a`.
#[test]
fn test_linear_operator_replace_indices_swap_is_simultaneous() {
    let a = make_index(2);
    let b = make_index(2);
    let op = build_identity_operator(&[("A".to_string(), a.clone()), ("B".to_string(), b.clone())]);

    let swapped = op
        .replace_indices(&[a.clone(), b.clone()], &[b.clone(), a.clone()])
        .unwrap();

    let input_true = |node: &str| {
        swapped
            .get_input_mapping(&node.to_string())
            .unwrap()
            .true_index
            .clone()
    };
    let output_true = |node: &str| {
        swapped
            .get_output_mapping(&node.to_string())
            .unwrap()
            .true_index
            .clone()
    };

    // Both sides reported the same true index (`a`) when the pairs were applied
    // one after another.
    assert_eq!(
        (input_true("A"), input_true("B")),
        (b.clone(), a.clone()),
        "input true indices must be swapped, not collapsed"
    );
    assert_eq!(
        (output_true("A"), output_true("B")),
        (b, a),
        "output true indices must be swapped, not collapsed"
    );
    assert_eq!(swapped.external_indices().len(), 4);
}

/// `Index` equality ignores `dim`, so an index that only shares id, tags and
/// plev with a stored true index must not match it: `replaceind` guarded this
/// with a dimension comparison and the batch path keeps that guard.
#[test]
fn test_linear_operator_replace_indices_keeps_dimension_guard() {
    let a = make_index(2);
    let op = build_identity_operator(&[("A".to_string(), a.clone())]);

    let mut aliased = a.clone();
    aliased.dim = 3;
    let target = make_index(3);

    let err = op.replace_indices(&[aliased], &[target]).unwrap_err();
    assert!(err.to_string().contains("not found"), "{err}");
}

/// Issue #1227's defect table lists both invalid batches as "(not checked)" for
/// `LinearOperator`. They are rejected now, before any mapping changes.
#[test]
fn test_linear_operator_replace_indices_rejects_duplicate_and_occupied_targets() {
    let a = make_index(2);
    let b = make_index(2);
    let op = build_identity_operator(&[("A".to_string(), a.clone()), ("B".to_string(), b.clone())]);

    // Two sources onto one target would leave two mappings with the same true
    // index, which is exactly the collapse the sequential path produced.
    let shared = make_index(2);
    let err = op
        .replace_indices(&[a.clone(), b.clone()], &[shared.clone(), shared.clone()])
        .unwrap_err();
    assert!(err.to_string().contains("Duplicate target"), "{err}");

    // `b` is a true index of this operator and is not replaced by the batch.
    let err = op
        .replace_indices(std::slice::from_ref(&a), std::slice::from_ref(&b))
        .unwrap_err();
    assert!(err.to_string().contains("already a true index"), "{err}");

    // The rejected batches left the operator untouched.
    assert_eq!(
        op.get_input_mapping(&"B".to_string()).unwrap().true_index,
        b
    );
}

#[test]
fn extending_a_complex_operator_across_a_steiner_gap_keeps_one_dtype() {
    use crate::operator::apply_linear_operator;
    use tensor4all_core::AnyScalar;

    // `i * I` on site0 and site2 of a four-site chain: site1 is a Steiner gap
    // bridging the operator's bond, site3 an outer gap. Only the site0 tensor
    // is complex, so the site2 tensor must be promoted as well.
    let (state, sites) = build_chain_state();
    let mut operator = build_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()]);
    let site0 = operator.mpo.node_index(&sites[0].0).unwrap();
    let complex_core = operator
        .mpo
        .tensor(site0)
        .unwrap()
        .scale(AnyScalar::new_complex(0.0, 1.0))
        .unwrap();
    operator.mpo.replace_tensor(site0, complex_core).unwrap();

    let (extended, _) = extend_operator_to_full_space(&operator, &state).unwrap();
    assert_eq!(extended.mpo.node_count(), state.node_count());
    for node in extended.mpo.node_indices() {
        assert!(
            extended.mpo.tensor(node).unwrap().is_complex(),
            "node {node:?} is real"
        );
    }

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let expected = state
        .to_dense()
        .unwrap()
        .scale(AnyScalar::new_complex(0.0, 1.0))
        .unwrap();
    let error = result
        .to_dense()
        .unwrap()
        .sub(&expected)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(error < 1e-12, "i * I differs by {error}");
}
