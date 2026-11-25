//! Exact local patch pairs retain their bonds even under a truncating policy.

use super::*;
use num_complex::Complex64;
use tensor4all_core::{SvdTruncationPolicy, TensorContractionLike, TensorElement};
use tensor4all_treetn::contraction::{self, ContractionMethod};

fn partition(data: TreeTN<IdxTensor, usize>) -> PartitionedTreeTN {
    PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(data).unwrap()).unwrap()
}

fn truncating(method: ContractionMethod) -> ContractionOptions {
    ContractionOptions {
        method,
        max_bond_dim: Some(1),
        svd_policy: Some(SvdTruncationPolicy::new(0.25)),
        ..Default::default()
    }
}

fn assert_dense(actual: &TreeTN<IdxTensor, usize>, expected: &IdxTensor) {
    let error = actual
        .to_dense()
        .unwrap()
        .sub(expected)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(error < 1e-11, "dense max absolute error: {error:e}");
}

fn local_pair<T: TensorElement>(value: impl Fn(f64) -> T) {
    // A branched tree also tests a bondless R=1 network through the same path.
    for nodes in [1, 4] {
        let inputs: Vec<_> = (0..nodes).map(|_| DynIndex::new_dyn(2)).collect();
        // Same logical IDs at different prime levels must remain output legs.
        let outputs: Vec<_> = inputs.iter().map(DynIndex::prime).collect();
        let state_bonds: Vec<_> = (1..nodes).map(|_| DynIndex::new_dyn(2)).collect();
        let op_bonds: Vec<_> = (1..nodes).map(|_| DynIndex::new_dyn(1)).collect();
        let state_tensors = (0..nodes)
            .map(|node| {
                let mut indices = vec![inputs[node].clone()];
                if node == 0 {
                    indices.extend(state_bonds.iter().cloned());
                } else {
                    indices.push(state_bonds[node - 1].clone());
                }
                let len: usize = indices.iter().map(|i| i.dim).product();
                let data = (0..len)
                    .map(|i| value((i + node + 1) as f64 / 7.0))
                    .collect();
                IdxTensor::from_dense(indices, data).unwrap()
            })
            .collect();
        let state = TreeTN::from_tensors(state_tensors, (0..nodes).collect()).unwrap();
        // Local signs, a general dense map, and a singular one-hot map. No
        // identity/unitarity/invertibility assumption is valid for this set.
        for local in [
            [1.0, 0.0, 0.0, -1.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 0.0],
        ] {
            let op_tensors = (0..nodes)
                .map(|node| {
                    let mut indices = vec![inputs[node].clone(), outputs[node].clone()];
                    if node == 0 {
                        indices.extend(op_bonds.iter().cloned());
                    } else {
                        indices.push(op_bonds[node - 1].clone());
                    }
                    IdxTensor::from_dense(indices, local.into_iter().map(&value).collect()).unwrap()
                })
                .collect();
            let operator = TreeTN::from_tensors(op_tensors, (0..nodes).collect()).unwrap();
            let expected = state
                .to_dense()
                .unwrap()
                .contract_pair(&operator.to_dense().unwrap())
                .unwrap();
            let left = partition(state.clone());
            let right = partition(operator);
            if nodes > 1 {
                // Standalone subdomain contraction keeps its previous policy;
                // only the ordinary partition pair loop selects exact pairs.
                let compressed = left
                    .values()
                    .next()
                    .unwrap()
                    .contract(
                        right.values().next().unwrap(),
                        &0,
                        truncating(ContractionMethod::Zipup),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(compressed.max_bond_dim(), 1);
            }
            for method in [
                ContractionMethod::Zipup,
                ContractionMethod::Fit,
                ContractionMethod::Src,
            ] {
                for (a, b) in [(&left, &right), (&right, &left)] {
                    for selected in [false, true] {
                        let result = if selected {
                            a.contract_selected(b, &0, truncating(method), &[Projector::new()])
                        } else {
                            a.contract(b, &0, truncating(method))
                        }
                        .unwrap();
                        let patch = result.get(&Projector::new()).unwrap();
                        assert_eq!(patch.data().link_dims(), vec![2; nodes - 1]);
                        assert!(patch.data().canonical_region().is_empty());
                        assert_dense(patch.data(), &expected);
                    }
                }
            }
        }
    }
}

#[test]
fn exact_unit_bond_arbitrary_local_tensors_real_and_complex() {
    local_pair(|x| x);
    local_pair(|x| Complex64::new(x, 0.2 * x));
}

#[test]
fn exact_unit_bond_three_leg_copy_tensor() {
    let shared = DynIndex::new_dyn(2);
    let out = shared.prime();
    let extra = DynIndex::new_dyn(2);
    let site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let unit = DynIndex::new_dyn(1);
    let state = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![shared.clone(), bond.clone()], vec![1.0, 2.0, 3.0, 5.0])
                .unwrap(),
            IdxTensor::from_dense(vec![bond, site], vec![2.0, 1.0, 4.0, 3.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let copy = IdxTensor::from_diag(vec![shared, out, extra], vec![2.0, -3.0]).unwrap();
    assert!(copy.is_diag());
    let op = TreeTN::from_tensors(
        vec![
            copy.contract_pair(&IdxTensor::from_dense(vec![unit.clone()], vec![1.0]).unwrap())
                .unwrap(),
            IdxTensor::from_dense(vec![unit], vec![1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let expected = state
        .to_dense()
        .unwrap()
        .contract_pair(&op.to_dense().unwrap())
        .unwrap();
    let result = partition(state)
        .contract(&partition(op), &0, truncating(ContractionMethod::Zipup))
        .unwrap();
    let patch = result.get(&Projector::new()).unwrap();
    assert_eq!(patch.data().link_dims(), vec![2]);
    assert_dense(patch.data(), &expected);
}

#[test]
fn exact_unit_bond_pairs_sum_exactly_then_truncate_once() {
    let row = DynIndex::new_dyn(2);
    let mid = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let unit = DynIndex::new_dyn(1);
    let state = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![row.clone(), mid.clone(), x, bond.clone()],
                vec![
                    1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 4.0, 3.0, 0.0, 1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 0.0,
                ],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, y], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let left = PartitionedTreeTN::from_subdomains(
        [(0, 0), (0, 1), (1, 0)]
            .into_iter()
            .map(|(r, m)| {
                SubDomainTreeTN::new(
                    state.clone(),
                    Projector::from_pairs([(row.clone(), r), (mid.clone(), m)]).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let right = partition(
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(vec![mid, unit.clone()], vec![1.0, -0.5]).unwrap(),
                IdxTensor::from_dense(vec![unit], vec![1.0]).unwrap(),
            ],
            vec![0, 1],
        )
        .unwrap(),
    );
    let options = truncating(ContractionMethod::Zipup);
    let pairs = left.contract_pairs(&right, &0, &options, None).unwrap();
    assert_eq!(pairs.len(), 3);
    for (a, b, c) in &pairs {
        assert_eq!(c.data().link_dims(), vec![2]);
        let expected = a
            .data()
            .to_dense()
            .unwrap()
            .contract_pair(&b.data().to_dense().unwrap())
            .unwrap();
        assert_dense(c.data(), &expected);
    }
    let keys: Vec<_> = (0..2)
        .map(|r| Projector::from_pairs([(row.clone(), r)]).unwrap())
        .collect();
    for engine in [SumEngine::Binary, SumEngine::Nary] {
        for selected in [false, true] {
            let opts = PartitionedContractionOptions {
                sum_engine: engine,
                ..options.clone().into()
            };
            let result = if selected {
                left.contract_selected(&right, &0, opts, &keys)
            } else {
                left.contract(&right, &0, opts)
            }
            .unwrap();
            let terms: Vec<_> = pairs
                .iter()
                .filter(|(_, _, c)| c.projector() == &keys[0])
                .map(|(_, _, c)| c)
                .collect();
            let mut expected = SubDomainTreeTN::exact_sum_checked(&terms, engine).unwrap();
            expected
                .truncate(&0, truncation_options_from_contraction(&options))
                .unwrap();
            let summed = result.get(&keys[0]).unwrap();
            assert_eq!(summed.max_bond_dim(), 1);
            assert_dense(summed.data(), &expected.data().to_dense().unwrap());
            let single = result.get(&keys[1]).unwrap();
            assert_eq!(single.max_bond_dim(), 2);
            let exact = pairs
                .iter()
                .find(|(_, _, c)| c.projector() == &keys[1])
                .unwrap();
            assert_dense(single.data(), &exact.2.data().to_dense().unwrap());
        }
    }
}

#[test]
fn exact_unit_bond_does_not_bypass_nonunit_pairs_or_option_validation() {
    let product = BlockProduct::new(None, 1.0);
    let options = truncating(ContractionMethod::Zipup);
    let pairs = product
        .a
        .contract_pairs(&product.b, &0, &options, None)
        .unwrap();
    assert_eq!(pairs.len(), BRANCHES.pow(3));
    for (a, b, c) in pairs {
        assert!(a.max_bond_dim() > 1 && b.max_bond_dim() > 1);
        let expected = contraction::contract(a.data(), b.data(), &0, options.clone()).unwrap();
        assert_eq!(c.data().link_dims(), expected.link_dims());
        assert_eq!(c.max_bond_dim(), 1);
        assert_dense(c.data(), &expected.to_dense().unwrap());
    }
    let site = DynIndex::new_dyn(2);
    let single = partition(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site], vec![1.0, 2.0]).unwrap()],
            vec![0],
        )
        .unwrap(),
    );
    for options in [
        ContractionOptions {
            max_bond_dim: Some(0),
            ..Default::default()
        },
        truncating(ContractionMethod::Naive),
    ] {
        assert!(matches!(
            single.contract(&single, &0, options),
            Err(PartitionedTreeTNError::InvalidOptions { .. })
        ));
    }
}

#[test]
fn exact_unit_bond_validates_method_specific_options_before_shortcuts() {
    let site = DynIndex::new_dyn(2);
    let single = partition(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site], vec![1.0, 2.0]).unwrap()],
            vec![0],
        )
        .unwrap(),
    );
    let invalid = [
        ContractionOptions {
            initial_max_bond_dim: Some(0),
            ..Default::default()
        },
        ContractionOptions {
            factorize_alg: tensor4all_core::FactorizeAlg::RSVD,
            ..Default::default()
        },
        ContractionOptions {
            method: ContractionMethod::Src,
            ..Default::default()
        },
    ];
    for options in invalid {
        assert!(matches!(
            options.validate(),
            Err(tensor4all_treetn::TreeTNOperationError::InvalidConfiguration { .. })
        ));
        for selected in [false, true] {
            let result = if selected {
                // Even no selected outputs must reject invalid options.
                single.contract_selected(&single, &0, options.clone(), &[])
            } else {
                single.contract(&single, &0, options.clone())
            };
            assert!(matches!(
                result,
                Err(PartitionedTreeTNError::TreeTN {
                    source: tensor4all_treetn::TreeTNOperationError::InvalidConfiguration { .. }
                })
            ));
        }
    }
}
